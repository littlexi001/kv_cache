# 01 — All-layer NoPE control

**Question.** Can content-only attention learn the long retrieval task, and what
local language-modeling ability is lost without any Q/K position rotation?

The phase is zero in every layer and frequency pair:

$$
\phi_{\ell i}(p)=0.
$$

This is a strong falsification control rather than the expected winner. Better
long retrieval with substantially worse DCLM PPL means position removal trades
away local order; it does not validate a generally useful PE. Failure on both
metrics shows that some position signal is required.

Config: `configs/strategies/all_nope.json`.
