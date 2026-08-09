# 00 — Native RoPE control

**Question.** What does ordinary Qwen3-0.6B pretraining learn under exactly the
same data, initialization, batch, optimizer, and token budget?

For every layer \(\ell\), frequency pair \(i\), and position \(p\), this control
uses

$$
\phi_{\ell i}(p)=p\omega_i.
$$

It is the required matched-compute baseline. A candidate passes only if its
long-context metric improves over this run at the same token checkpoint while
its held-out DCLM PPL stays within the declared guardrail. The untouched random
initialization is not a sufficient baseline.

Config: `configs/strategies/native_rope.json`. Inspect
`strategy_profile.json`, TensorBoard loss/LR/gradient norm, and checkpoint
evaluation summaries.
