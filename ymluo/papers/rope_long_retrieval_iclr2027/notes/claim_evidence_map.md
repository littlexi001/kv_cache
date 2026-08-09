# Claim–evidence map

Protocol note: the analytic NF4 phase case, the position-extended BF16 age
trajectory, the native-window BF16 population audit, and the held-out NF4
retrieval diagnostic are distinct regimes. Evidence can be triangulated across
them, but no same-example full-path mediation proportion is claimed.

| Paper claim | Current evidence | Strength | Required follow-up |
|---|---|---|---|
| Fixed semantic Q/K can receive different post-RoPE scores as distance changes | Identical layer-0 pre-RoPE Q/K at four distances; 64-pair reconstruction error ≤ 0.00198 | Exact for the audited Qwen3-8B case | Repeat across seeds, models, and native windows |
| A score perturbation changes the attention output through a centered Value write | Exact derivative: `∂o/∂s_j = a_j W_O(v_j - v̄)` | Analytic identity | None for the local identity |
| Answer-directed score utility predicts the finite answer-margin effect of a selected evidence score | No-op-matched BF16 singleton interventions at 8K/16K/24K; evidence Pearson 0.960/0.973/0.936 | Strong local causal closure in the instrumented Qwen3-8B graph | Second model, natural tasks, and non-oracle target protocols |
| No reliable seed-macro gold−conflict mean gap is detected for attention probability or grid-envelope RoPE suppression | Seed-aggregated gold−conflict intervals cross zero at 8K/16K/24K | Controlled mean-gap result; not equivalence or token-level nonclassification | Counterbalanced natural conflict data and equivalence/classification tests |
| Intermediate residual divergence can switch the answer | Exact 36-layer replay plus residual patching at L16/L20 in one transition | Strong single-case causal evidence | Population activation patching with negative controls |
| Query-specific phase repair is functionally a logit edit | Exact algebra under fixed support and Values | Theorem-level, scoped | State scope explicitly; do not generalize to all geometries |
| One reusable GQA cache realizes a phase repair iff corrections are Query/Key separable | Four-cycle equivalence theorem under one-basis phase-only SO(2) assumptions | Theorem-level, scoped | Optional numerical construction tests |
| The full phase→denominator→Value→later-Query→margin mediation proportion is known | Not established | Unsupported | Registered component and mediation interventions |
| Pre-RoPE proposal is a novel deployable method or accelerator | Direct prior art and full-history scan | Unsupported | Keep only as diagnostic baseline unless a distinct indexed system is developed |
| The mechanism is universal across RoPE LLMs | One model family and controlled tasks | Unsupported | Cross-model, public-task, native-window full-chain activation patching |
