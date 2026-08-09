# 02 — Periodic RoPE/NoPE layers

**Prior.** RoPE layers can learn local order while interleaved NoPE layers offer
a content-only route for remote matching. This tests the layer-mixing axis
studied by [Rope to Nope and Back Again](https://arxiv.org/abs/2501.18795),
using an independently specified implementation.

For zero-based layer index \(\ell\):

$$
\phi_{\ell i}(p)=
\begin{cases}
0,& \ell\bmod 4=3,\\
p\omega_i,&\text{otherwise}.
\end{cases}
$$

It passes only if retrieval improves over native RoPE without a material PPL
penalty. Inspect whether gains occur consistently across positions; a gain only
at one synthetic length is insufficient.

Config: `configs/strategies/rnope_every4.json`.
